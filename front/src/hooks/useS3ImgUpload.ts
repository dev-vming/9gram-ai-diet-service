import useApi from './useApi';

type S3ImgUploadProps = {
  file?: File | null;
  presignedUrl: string;
};

const useS3ImgUpload = () => {
  const { trigger, result, error, loading } = useApi({
    shouldInitFetch: false,
  });

  const uploadToS3 = async ({ presignedUrl, file }: S3ImgUploadProps) => {
    if (!file)
      return {
        error: '업로드할 파일이 없습니다',
        loading: false,
      };

    await trigger({
      method: 'put',
      path: presignedUrl,
      data: file,
    });

    if (error) {
      return { error, loading: false };
    }
    return { message: '이미지 업로드 성공' };
  };

  return { uploadToS3 };
};

export default useS3ImgUpload;
