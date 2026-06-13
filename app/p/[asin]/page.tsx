import HomeClient from '@/app/HomeClient';
import { normalizeAsinInput } from '@/lib/shareRoutes';

type ProductSharePageProps = {
  params: {
    asin: string;
  };
};

export default function ProductSharePage({ params }: ProductSharePageProps) {
  return <HomeClient initialAsin={normalizeAsinInput(params.asin)} />;
}
