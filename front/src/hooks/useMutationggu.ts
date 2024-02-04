import {
  MutationFunction,
  QueryClient,
  QueryKey,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query';
import { ApiMethods } from '@utils/axiosConfig';
import { isAxiosError } from 'axios';
import { AxiosError } from 'axios';
import { useErrorBoundary } from 'react-error-boundary';

type ApiResponseError = AxiosError;

const useMutationggu = <T, E = ApiResponseError>(
  itemId: QueryKey,
  trigger: MutationFunction<T>,
  gcTime: number,
  applyResult: boolean,
  isShowBoundary: boolean,
  method: ApiMethods
) => {
  const queryClient = useQueryClient();
  const { showBoundary } = useErrorBoundary();

  return useMutation<T, E, any>({
    mutationFn: trigger,
    //method get이고 gcTime적용해야 할때는 기존 값을 반환해서 캐싱을 적용해야 함
    mutationKey: itemId,
    onMutate: async () => {
      if (method === 'get' && gcTime > 0) {
        // await queryClient.cancelQueries({ queryKey: itemId });

        // 기존 Query를 가져오는 함수 ( 존재하지 않으면 undefind 반환 )
        const previousValue = queryClient.getQueryData(itemId);

        //optimistic UI 현재는 필요없음
        // if (previousValue) {
        //   // setQueryData(): Query의 캐시된 데이터를 즉시 업데이트하는 동기 함수 ( Query가 존재하지 않으면 생성 )
        //   // 전달받은 variables값을 즉시 새로운 데이터로 업데이트
        //   queryClient.setQueryData(itemId, (oldData) => [...oldData, variables]);
        // }

        // 이전 값 리턴
        return { previousValue };
      }
    },

    onSuccess: async (data) => {
      if (applyResult && gcTime !== 0) {
        queryClient.invalidateQueries({ queryKey: itemId });
        queryClient.setQueriesData({ queryKey: itemId }, data);
        return;
      }
      return;
    },
    onError: async (error, variables, context) => {
      if (isAxiosError(error)) {
        console.log(error.message);
      }
      isShowBoundary && showBoundary(error);
    },
    onSettled: async () => {},
    throwOnError: isShowBoundary,
    gcTime,
    //gcTime만 해도 될지? staleTime은 refetch의 기능이므로 필요 없을지?
  });
};

export default useMutationggu;
