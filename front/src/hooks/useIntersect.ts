import { useEffect, useState } from 'react';

interface useIntersectProps {
  root?: null;
  rootMargin?: string;
  threshold?: number;
  onIntersect: IntersectionObserverCallback;
}

const useIntersect = ({
  root,
  rootMargin,
  threshold,
  onIntersect,
}: useIntersectProps) => {
  const [target, setTarget] = useState<HTMLElement|null>(null);

  useEffect(() => {
    if (!target) return;

    const observer: IntersectionObserver = new IntersectionObserver(
      onIntersect,
      { root, rootMargin, threshold }
    );

    observer.observe(target);


    return () => observer.unobserve(target);
  }, [onIntersect, root, rootMargin, target, threshold]);

  return { setTarget };

};

export default useIntersect;
