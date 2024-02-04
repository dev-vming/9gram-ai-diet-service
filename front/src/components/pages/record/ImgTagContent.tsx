import { ImgTagContentProps } from './RecordTypes';
import { MergingTags } from './MergingTags';

const ImgTagContent: React.FC<ImgTagContentProps> = ({
  imgUrl,
  className,
  data,
  selectedMealNumber,
}) => {
  const tagData = data?.[selectedMealNumber].foods;

  return (
    <div style={{ position: 'relative' }}>
      {imgUrl && <img className={className} src={imgUrl} alt='식단 이미지' />}
      <MergingTags tagData={tagData} />
    </div>
  );
};

export default ImgTagContent;
