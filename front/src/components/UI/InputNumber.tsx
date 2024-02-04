import { useId, useState } from 'react';
import Down from '../../assets/Down';
import Up from '../../assets/Up';
import classes from './InputNumber.module.css';
import { useControlled } from '@hooks/useControlled';
import Toast from './Toast';
import ToastText from './ToastText';

interface InputBaseProps {
  label?: string;
  type?: string;
  id?: string;
  defaultValue?: string;
  value?: number;
  className?: string | {};
  placeholder?: string;
  required?: false;
  disabled?: false;
  onValueChange?: (value: any) => void;
  checkAll?: false;
  maximumValue?: number;
}

const InputNumber = (props: InputBaseProps) => {
  const { onValueChange, defaultValue = +1, value, maximumValue = +31 } = props;
  const [inputNum, setInputNum] = useControlled({
    controlled: value,
    unControlled: defaultValue,
  });
  const [showToast, setShowToast] = useState(false);

  const controlledId = props.id;
  const uncontrolledId = useId();
  const id = useControlled({
    controlled: controlledId,
    unControlled: uncontrolledId,
  })[0];

  const handleInputFn = (
    e: React.ChangeEvent<HTMLInputElement>,
    operator: 'minus' | 'sum'
  ) => {
    let newInputNum: number = 1;
    operator === 'minus' &&
      (newInputNum = inputNum - 1 === 0 ? maximumValue : inputNum - 1);
    operator === 'sum' &&
      (newInputNum = inputNum === maximumValue ? 1 : inputNum + 1);
    const showToast =
      operator === 'minus' ? newInputNum < 1 : newInputNum > maximumValue;

    setShowToast(showToast);
    setInputNum(newInputNum);
    onValueChange?.(newInputNum);
  };

  const onChangeHandler = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = Number(e.target.value);
    let newValue;
    if (!inputValue || inputValue < 1 || inputValue > maximumValue) {
      setShowToast(true);
      newValue = inputValue <= 0 ? 1 : maximumValue;
    } else {
      newValue = Math.max(inputValue, 1);
    }
    setInputNum(newValue);
    onValueChange?.(newValue);
  };

  return (
    <>
      <div className={classes.numberInput}>
        <input
          type='number'
          id={id}
          value={inputNum}
          className={classes.input}
          onChange={onChangeHandler}
          max={maximumValue}
        />
        <div className={classes.absolute}>
          <div
            className={classes.upper}
            onClick={(e: any) => {
              handleInputFn(e, 'sum');
            }}
          >
            <Up />
          </div>
          <div
            className={classes.down}
            onClick={(e: any) => {
              handleInputFn(e, 'minus');
            }}
          >
            <Down />
          </div>
        </div>
        <Toast show={showToast} setShow={setShowToast}>
          <ToastText>올바른 값을 입력해주세요 1 ~ {maximumValue}</ToastText>
        </Toast>
      </div>
    </>
  );
};

export default InputNumber;
