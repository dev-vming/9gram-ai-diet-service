import { forwardRef, useId, useImperativeHandle, useRef } from 'react';
import { useControlled } from '@hooks/useControlled';
import classes from './inputCommon.module.css';
import { getClassNamesArr } from '@utils/getClassesArr';
import { InputVariant, GetClassNameType } from 'typings/commontypes';
import { CommonSizeType } from 'typings/commontypes';

type InputPropsType = {
  value?: string | number;
  defaultValue?: string | number | readonly string[];
  className?: string;
  onBlur?: any;
  placeholder?: string;
  disabled?: boolean;
  required?: boolean;
  type?: string;
  onChange?: (value: React.ChangeEvent<HTMLInputElement>) => void;
  id?: string;
  variant?: InputVariant;
  active?: boolean;
  size?: CommonSizeType;
  style?: {};
};
const getModuleClassName = (
  customClassName: string | undefined,
  { variant, disabled, active, size, prefix }: GetClassNameType
) => {
  //get = ['variant-active', '', 'active-true', 'size-large']
  const get = getClassNamesArr(
    customClassName,
    Object.entries({ variant, disabled, active, size })
  );
  if (typeof get === 'object' && get.length > 0) {
    //prefix 스타일은 button, 중간은 classes 하이픈고려, custom은 마지막에 적용
    //`input ${classes[`${variant-active}`]} ${classes[`${active-true}`]} ... customClassName`
    const returnedClasses =
      `${classes[`${prefix}`]}` +
        ' ' +
        get.map((classN) => classN && `${classes[`${classN}`]}`).join(' ') +
        ' ' +
        customClassName || '';
    return returnedClasses;
  } else if (get?.length === 0) {
    return prefix + ' ' + customClassName && customClassName;
  }
};
//inputCommon에 value를 넣으면 controlled. 기본은 uncontrolled
//uncontrolled일때만 input을 자동 변경하게 해줌
const InputCommon = forwardRef(
  (
    {
      value,
      defaultValue,
      className,
      variant = 'default',
      disabled,
      active,
      size = 'large',
      ...props
    }: InputPropsType,
    ref
  ) => {
    const inputRef = useRef<HTMLInputElement>(null);

    const [input, setInput] = useControlled({
      controlled: value,
      unControlled: defaultValue,
    });

    const controlledId = props.id;
    const uncontrolledId = useId();

    const id = useControlled({
      controlled: controlledId,
      unControlled: uncontrolledId,
    })[0];

    const inputClass = getModuleClassName(className, {
      variant,
      disabled,
      active,
      size,
      prefix: 'input',
    });

    return (
      <>
        {onchange ? (
          <input
            id={id}
            value={input}
            onChange={(e) => setInput(e)}
            className={inputClass}
            autoComplete='off'
            {...props}
          />
        ) : (
          <input
            defaultValue={defaultValue}
            ref={inputRef}
            autoComplete='off'
            {...props}
            className={inputClass}
          />
        )}
      </>
    );
  }
);

export default InputCommon;
