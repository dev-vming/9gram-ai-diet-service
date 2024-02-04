import { useCallback } from 'react';
import useApi from './useApi';

type presignedUrlProps = {
  fileName?: string;
  path: any;
};

const usePresignedUrl = ({ fileName, path }: presignedUrlProps) => {
  const { trigger, result, error, loading } = useApi({
    method: 'post',
    shouldInitFetch: false,
  });

  const getPresignedUrl = useCallback(
    async ({ fileName, path }: presignedUrlProps) => {
      await trigger({ path: path, data: { fileName } });
    },
    [fileName]
  );

  return {
    getPresignedUrl,
    presignedUrl: result as { data: string },
    error,
    loading,
  };
};

export default usePresignedUrl;
