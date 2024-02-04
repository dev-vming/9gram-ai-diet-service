import styles from './deletetag.module.css';

interface Props {
  tags: string[];
  setTags: React.Dispatch<React.SetStateAction<string[]>>;
}

const DeleteTag = ({ tags, setTags }: Props) => {
  const deleteTag = (e: React.MouseEvent<HTMLSpanElement, MouseEvent>) => {
    if (!e.currentTarget.parentNode) {
      throw new Error('삭제할 태그 내용이 없습니다!');
    }
    const selectTagText = e.currentTarget.parentNode.textContent;
    const newTags = tags.filter((tag) => tag !== selectTagText);
    setTags(newTags);
  };

  return (
    <>
      {tags.map((tag, index) => (
        <span key={index} className={styles.tag}>
          {tag}
          <img
            className={styles.xbtn}
            src='/icons/x-icon.png'
            alt='태그삭제'
            onClick={(e) => deleteTag(e)}
          />
        </span>
      ))}
    </>
  );
};

export default DeleteTag;
