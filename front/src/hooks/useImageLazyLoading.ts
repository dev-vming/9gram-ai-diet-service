import { useEffect, useRef, useState } from 'react';

export const useImageLazyLoading = () => {
  const targetRef = useRef<HTMLImageElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  const handler = ([entry]: IntersectionObserverEntry[]) => {
    if (entry.isIntersecting) {
      setIsLoaded(true);
    }
  };

  useEffect(() => {
    const observer = new IntersectionObserver(handler, {
      threshold: 1,
      rootMargin: '20px',
    });
    targetRef.current && observer.observe(targetRef.current);

    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetRef, isLoaded]);

  return { targetRef, isLoaded };
};
