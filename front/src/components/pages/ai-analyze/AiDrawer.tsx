import styles from '@components/pages/ai-analyze/drawer.module.css';
import useCachingApi from '@hooks/useCachingApi';
import useIntersect from '@hooks/useIntersect';
import { useEffect, useState } from 'react';
import DrawerCard from './DrawerCard';

interface DrawerList {
  feedbackId: string;
  feedbackDate: string;
  questionType: string;
  question: string;
  feedback: string;
}
type DrawerApi = {
  data: { data: DrawerList[] };
};

const AiDrawer = () => {
  const [drawerList, setDrawerList] = useState<DrawerList[]>([]);
  const [page, setPage] = useState(1);
  const [isFirst, setIsFirst] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  const { trigger, result } = useCachingApi<DrawerApi>({
    path: `/feedback?page=${page}`,
  });

  useEffect(() => {
    if (!isFirst) return;
    trigger('', {
      onSuccess: (data) => {
        setDrawerList((prev) => [...prev, ...data.data.data]);
      },
    });
    setIsFirst(false);
  }, []);

  const onIntersect: IntersectionObserverCallback = async (
    [entry],
    observer
  ) => {
    if (entry.isIntersecting) {
      setIsLoading(true);
      observer.unobserve(entry.target);
      trigger('', {
        onSuccess: (data) => {
          setDrawerList((prev) => [...prev, ...data.data.data]);
          setIsLoading(false);
        },
      });
      observer.observe(entry.target);
      setPage((prev) => prev + 1);
    }
  };

  const { setTarget } = useIntersect({
    root: null,
    rootMargin: '100px',
    threshold: 1,
    onIntersect,
  });

  const handleDelete = (deletedId: string) => {
    setDrawerList(
      drawerList.filter((drawer) => drawer.feedbackId !== deletedId)
    );
  };

  return (
    <div className={styles.drawer_wrapper}>
      {drawerList?.map((drawer, idx) => (
        <div key={`drawercard-${idx}`}>
          <DrawerCard
            id={drawer.feedbackId}
            date={drawer.feedbackDate}
            type={drawer.questionType}
            tag={drawer.question}
            text={drawer.feedback}
            onDeleted={handleDelete}
          />
          {!isLoading && <div ref={setTarget} />}
        </div>
      ))}
    </div>
  );
};

export default AiDrawer;
