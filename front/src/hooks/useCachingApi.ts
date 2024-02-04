import { useEffect } from 'react';
import { API_FETCHER, ApiMethods } from '@utils/axiosConfig';
import useMutationggu from './useMutationggu';

interface UseApiParams {
  method?: ApiMethods;
  path: string;
  data?: any;
  shouldInitFetch?: boolean;
  initialResult?: string;
  gcTime?: number;
  applyResult?: boolean;
  isShowBoundary?: boolean;
}

const useCachingApi = <T>({
  method: triggerMethod = 'get',
  path: triggerPath = '',
  data: triggerData = {},
  shouldInitFetch = false,
  applyResult = true,
  isShowBoundary = true,
  // initialResult = '',
  gcTime = 0,
}: UseApiParams) => {
  // }) => {
  const key = triggerPath;

  const keyArr: string[] = [];
  keyArr.push(key);

  const {
    mutate,
    data,
    isPending: loading,
    error,
  } = useMutationggu<T>(
    keyArr,
    async (data = triggerData) =>
      await API_FETCHER[triggerMethod as ApiMethods](triggerPath, data),
    gcTime,
    applyResult,
    isShowBoundary,
    triggerMethod
  );

  useEffect(() => {
    shouldInitFetch && mutate(triggerPath, triggerData);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reqIdentifier = triggerMethod + 'data';

  return {
    result: data,
    loading,
    reqIdentifier,
    trigger: mutate,
    error,
  };
};

export default useCachingApi;
