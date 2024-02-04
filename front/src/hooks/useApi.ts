import { useEffect, useState, useCallback } from 'react';
import { API_FETCHER, ApiMethods } from '@utils/axiosConfig';
import { useErrorBoundary } from 'react-error-boundary';
import { AxiosError, AxiosResponse } from 'axios';
import axios from 'axios';

interface UseApiParams {
  method?: ApiMethods;
  path?: string;
  data?: any;
  shouldInitFetch?: boolean;
  initialResult?: any;
}
interface TriggerPropsType
  extends Omit<UseApiParams, 'shouldInitFetch' | 'initialResult'> {
  applyResult?: boolean;
  isShowBoundary?: boolean;
}

export type TriggerType = ({
  ...props
}: TriggerPropsType) => Promise<AxiosResponse<any, any>>;

const useApi = <T>({
  method = 'get',
  path = '',
  data = {},
  shouldInitFetch = false,
  initialResult = '',
}: UseApiParams): {
  result: T | undefined;
  loading: boolean;
  reqIdentifier: string;
  trigger: TriggerType;
  error: any;
} => {
  const [result, setResult] = useState(initialResult);
  const [loading, setLoading] = useState(false);
  const [reqIdentifier, setReqIdentifier] = useState('');
  const [error, setError] = useState({});
  const { showBoundary } = useErrorBoundary();

  const trigger: TriggerType = useCallback(
    async ({
      method: triggerMethod = method,
      path: triggerPath = path,
      data: triggerData = data,
      applyResult = true,
      isShowBoundary = true,
    }) => {
      setLoading(true);
      setReqIdentifier(triggerMethod + 'Data');
      try {
        const triggerResult = await API_FETCHER[triggerMethod as ApiMethods](
          triggerPath,
          triggerData
        );

        if (applyResult) {
          setResult(triggerResult);
          return;
        }
        return triggerResult;
      } catch (err) {
        console.log(err);
        if (axios.isAxiosError(err) && isShowBoundary) {
          //에러 바운더리를 보여줘야 할때만 보여줌
          showBoundary(err);
          return;
        }
        axios.isAxiosError(err) && setError(err);
        return;
      } finally {
        setLoading(false);
      }
    },
    [data, method, showBoundary, path, result]
  );

  useEffect(() => {
    shouldInitFetch && console.log('초기 요청합니다!!', method, path);
    shouldInitFetch && trigger({ method, path, data });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { result, loading, reqIdentifier, trigger, error };
};
export default useApi;
