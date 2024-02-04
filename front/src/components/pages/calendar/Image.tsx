import { useImageLazyLoading } from '@hooks/useImageLazyLoading';

interface ImagePropsType {
  src: string;
}
const Image = ({ src }: ImagePropsType) => {
  const { targetRef, isLoaded } = useImageLazyLoading();

  return (
    <img
      ref={targetRef}
      src={isLoaded ? src : ''}
      style={isLoaded ? { backgroundColor: 'gray' } : {}}
      alt={src}
    />
  );
};

export default Image;
