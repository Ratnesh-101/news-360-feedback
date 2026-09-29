import React, { useState, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { UploadCloud, Mic, Play, Pause, CheckCircle2, AlertCircle, RefreshCw, FileAudio } from 'lucide-react';
import { api } from '@/lib/api/endpoints';
import { SentimentBadge } from '@/components/SentimentBadge';
import { ScoreMeter } from '@/components/ScoreMeter';
import type { AudioUpload } from '@/types/domain';

export default function AudioStudioPage() {
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobData, setJobData] = useState<AudioUpload | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isApproving, setIsApproving] = useState(false);
  const [approvedSuccess, setApprovedSuccess] = useState(false);

  // Poll active audio job until ready or failed
  useEffect(() => {
    if (!jobId || jobData?.status === 'ready' || jobData?.status === 'indexed' || jobData?.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await api.getAudioJob(jobId);
        setJobData(updated);
      } catch (e) {
        console.error('Failed to poll audio job:', e);
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [jobId, jobData?.status]);

  const handleFileUpload = async (file: File) => {
    try {
      setIsUploading(true);
      setApprovedSuccess(false);
      const res = await api.uploadAudio(file);
      setJobId(res.id);
      const initial = await api.getAudioJob(res.id);
      setJobData(initial);
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleApprove = async () => {
    if (!jobId) return;
    try {
      setIsApproving(true);
      await api.approveAudioJob(jobId);
      setApprovedSuccess(true);
      // Invalidate relevant queries so dashboard updates
      queryClient.invalidateQueries({ queryKey: ['overview'] });
      queryClient.invalidateQueries({ queryKey: ['articles'] });
      queryClient.invalidateQueries({ queryKey: ['ministries'] });
    } catch (err: any) {
      alert(`Approval failed: ${err.message}`);
    } finally {
      setIsApproving(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <header>
        <h1 className="text-2xl font-serif font-bold text-ink">Broadcast & Audio News Studio</h1>
        <p className="text-xs text-ink/60 mt-1">
          Upload regional spoken news bulletins (Hindi, Marathi, English) for automated Whisper transcription, translation, and sentiment extraction
        </p>
      </header>

      {/* Upload Dropzone */}
      <div className="rounded-panel border-2 border-dashed border-rule bg-surface p-8 text-center space-y-4 hover:border-chakra/60 transition-colors">
        <div className="mx-auto size-12 rounded-full bg-chakra/10 text-chakra flex items-center justify-center">
          <UploadCloud className="size-6" />
        </div>

        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-ink">Drag and drop audio broadcast clips</h3>
          <p className="text-xs text-ink/50">Supports MP3, WAV, M4A, OGG, FLAC up to 50MB</p>
        </div>

        <div className="flex items-center justify-center gap-3">
          <label className="cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-control bg-chakra text-white text-xs font-semibold hover:bg-chakra/90 transition-colors shadow-sm">
            <FileAudio className="size-4" />
            <span>Select Audio File</span>
            <input
              type="file"
              accept=".mp3,.wav,.m4a,.ogg,.flac"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFileUpload(f);
              }}
            />
          </label>

          <button
            onClick={() => {
              // Demo audio sample
              const blob = new Blob(['sample audio binary content'], { type: 'audio/mp3' });
              const file = new File([blob], 'bullet_train_marathi_report.mp3', { type: 'audio/mp3' });
              handleFileUpload(file);
            }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-control border border-rule hover:bg-paper text-xs text-ink font-semibold transition-colors"
          >
            <Mic className="size-4 text-chakra" />
            <span>Load Sample Regional Clip</span>
          </button>
        </div>
      </div>

      {/* Processing State */}
      {isUploading || (jobData && jobData.status !== 'ready' && jobData.status !== 'indexed') ? (
        <div className="rounded-panel border border-rule bg-surface p-6 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-ink flex items-center gap-2">
              <RefreshCw className="size-3.5 animate-spin text-chakra" />
              Processing Pipeline: {jobData?.status ? jobData.status.toUpperCase() : 'UPLOADING'}
            </span>
            <span className="font-mono text-ink/50">{Math.round((jobData?.progress || 0.1) * 100)}%</span>
          </div>
          <div className="h-2 w-full bg-rule/40 rounded-full overflow-hidden">
            <div
              className="h-full bg-chakra transition-all duration-500 rounded-full"
              style={{ width: `${Math.round((jobData?.progress || 0.1) * 100)}%` }}
            />
          </div>
        </div>
      ) : null}

      {/* Audio Result & Verification Studio */}
      {jobData?.preview && (
        <div className="rounded-panel border border-rule bg-surface shadow-sm overflow-hidden space-y-6 p-6">
          {/* Audio Player Strip */}
          <div className="rounded-control border border-rule bg-paper p-4 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="size-10 rounded-full bg-chakra text-white flex items-center justify-center hover:bg-chakra/90 transition-colors shadow-sm"
              >
                {isPlaying ? <Pause className="size-4" /> : <Play className="size-4 ml-0.5" />}
              </button>
              <div>
                <h4 className="text-sm font-semibold text-ink">{jobData.filename}</h4>
                <p className="text-[11px] font-mono text-ink/50">Language detected: {jobData.language?.toUpperCase()} (Whisper Base)</p>
              </div>
            </div>

            {/* Mock Waveform Visualizer */}
            <div className="hidden sm:flex items-center gap-1 h-6">
              {[40, 70, 25, 90, 60, 85, 30, 95, 75, 50, 80, 45, 65, 35, 90, 70, 40].map((h, i) => (
                <div
                  key={i}
                  className={`w-1 rounded-full transition-all duration-200 ${
                    isPlaying ? 'bg-chakra animate-pulse' : 'bg-rule'
                  }`}
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
          </div>

          {/* Synchronized Dual Transcript */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-ink/60">
                Original Spoken Transcript (Regional)
              </h4>
              <div className="p-4 rounded-control bg-paper/60 border border-rule/60 text-sm leading-relaxed font-serif text-ink min-h-[140px]">
                {jobData.segments.map((seg, i) => (
                  <p key={i} className="mb-2">
                    {seg.original}
                  </p>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-ink/60">
                English Translation (Analyzed)
              </h4>
              <div className="p-4 rounded-control bg-paper/60 border border-rule/60 text-sm leading-relaxed text-ink min-h-[140px]">
                {jobData.segments.map((seg, i) => (
                  <p key={i} className="mb-2">
                    {seg.translated}
                  </p>
                ))}
              </div>
            </div>
          </div>

          {/* Tag & Sentiment Inspector */}
          <div className="p-4 rounded-control border border-rule/80 bg-paper/30 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <SentimentBadge value={jobData.preview.sentiment} score={jobData.preview.sentimentScore} />
                <span className="text-xs font-semibold text-chakra">{jobData.preview.ministry}</span>
              </div>
              <ScoreMeter score={jobData.preview.sentimentScore} sentiment={jobData.preview.sentiment} />
            </div>

            <p className="text-xs text-ink/80 leading-relaxed">
              <strong>Assessment Reason:</strong> {jobData.preview.reason}
            </p>

            <div className="flex flex-wrap gap-1.5 pt-1">
              {jobData.preview.keywords.map((kw, i) => (
                <span key={i} className="px-2 py-0.5 rounded text-[11px] bg-rule/40 font-mono text-ink/70">
                  #{kw}
                </span>
              ))}
            </div>
          </div>

          {/* Action Strip */}
          <div className="flex items-center justify-between pt-2">
            {approvedSuccess ? (
              <div className="flex items-center gap-2 text-pos text-xs font-semibold">
                <CheckCircle2 className="size-4" />
                <span>Indexed into Live Articles Database and FAISS Vector Store</span>
              </div>
            ) : (
              <span className="text-xs text-ink/50 font-mono">
                Status: Verified and ready for knowledge-base indexing
              </span>
            )}

            <button
              onClick={handleApprove}
              disabled={isApproving || approvedSuccess}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-control bg-chakra text-white text-xs font-semibold hover:bg-chakra/90 disabled:opacity-50 transition-colors shadow-sm"
            >
              {isApproving ? <RefreshCw className="size-3.5 animate-spin" /> : <CheckCircle2 className="size-3.5" />}
              <span>{approvedSuccess ? 'Approved & Indexed' : 'Approve & Index to Vector DB'}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
