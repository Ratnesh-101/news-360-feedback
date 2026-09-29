import { createBrowserRouter } from 'react-router-dom';
import { AppShell } from './layout/AppShell';
import DashboardPage from '@/features/dashboard/DashboardPage';
import MinistryLeaderboard from '@/features/ministries/MinistryLeaderboard';
import MinistryDetailPage from '@/features/ministries/MinistryDetailPage';
import AudioStudioPage from '@/features/audio/AudioStudioPage';
import BriefingPage from '@/features/briefings/BriefingPage';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'ministries', element: <MinistryLeaderboard /> },
      { path: 'ministries/:slug', element: <MinistryDetailPage /> },
      { path: 'audio', element: <AudioStudioPage /> },
      { path: 'briefing', element: <BriefingPage /> },
    ],
  },
]);
