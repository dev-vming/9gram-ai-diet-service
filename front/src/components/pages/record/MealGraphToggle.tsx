import style from './mealgraphtoggle.module.css';
interface MealGraphTogglePropsType {
  isShowingTotal: boolean;
  setIsShowingTotal: (value: boolean) => void;
}

const MealGraphToggle = ({
  isShowingTotal,
  setIsShowingTotal,
}: MealGraphTogglePropsType) => {
  const onClickBtn = () => {
    setIsShowingTotal(!isShowingTotal);
  };

  return (
    <div
      className={`${style.toggleButton} ${isShowingTotal ? '' : style.toggleButtonActive}`}
      onClick={onClickBtn}
    >
      <div></div>
    </div>
  );
};
export default MealGraphToggle;
