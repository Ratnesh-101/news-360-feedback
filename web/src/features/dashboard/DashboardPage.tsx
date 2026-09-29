import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api/endpoints';
import { useFilters } from '@/stores/filters';
import { KpiGrid } from './components/KpiGrid';
import { SentimentTimeChart } from './components/SentimentTimeChart';
import { DistributionChart } from './components/DistributionChart';
import { AlertsPanel } from './components/AlertsPanel';
import { NewsFeed } from './components/NewsFeed';

export default function DashboardPage() {
  const getApiParams = useFilters((s) => s.getApiParams);
  const filterParams = getApiParams();

  const overviewQuery = useQuery({
    queryKey: ['overview', filterParams],
    queryFn: () => api.overview(filterParams),
    refetchInterval: 30_000,
  });

  const timeseriesQuery = useQuery({
    queryKey: ['timeseries', filterParams],
    queryFn: () => api.timeseries({ bucket: 'day', ...filterParams }),
  });

  const distributionQuery = useQuery({
    queryKey: ['distribution'],
    queryFn: () => api.distribution('category'),
  });

  const alertsQuery = useQuery({
    queryKey: ['alerts'],
    queryFn: () => api.alerts(),
    refetchInterval: 60_000,
  });

  const articlesQuery = useQuery({
    queryKey: ['articles', filterParams],
    queryFn: () => api.articles({ ...filterParams, limit: 30 }),
  });

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <KpiGrid data={overviewQuery.data} isLoading={overviewQuery.isLoading} />

      {/* Active Attention Alerts */}
      <AlertsPanel alerts={alertsQuery.data} isLoading={alertsQuery.isLoading} />

      {/* Visual Analytics */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SentimentTimeChart data={timeseriesQuery.data} isLoading={timeseriesQuery.isLoading} />
        </div>
        <div>
          <DistributionChart data={distributionQuery.data} isLoading={distributionQuery.isLoading} />
        </div>
      </div>

      {/* Real-time Filterable Feed */}
      <NewsFeed
        articles={articlesQuery.data?.items}
        total={articlesQuery.data?.total}
        isLoading={articlesQuery.isLoading}
      />
    </div>
  );
}
