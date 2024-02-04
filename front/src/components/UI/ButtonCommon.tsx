import React, { useId, forwardRef, useRef } from 'react';
import classes from './buttonCommon.module.css';
import { useButtonProps } from '@hooks/useButtonProps';
import { getClassNamesArr } from '@utils/getClassesArr';
import { ButtonPropsType, GetClassNameType } from 'typings/commontypes';

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
    //`button ${classes[`${variant-active}`]} ${classes[`${active-true}`]} ... customClassName`
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

const ButtonCommon = forwardRef<HTMLButtonElement, ButtonPropsType>(
  (
    {
      children,
      onClickBtn,
      customClassName,
      size,
      disabled = false,
      active = true,
      variant = 'default',
      href,
      ...props
    }: ButtonPropsType,
    ref
  ) => {
    const buttonRef = useRef<HTMLButtonElement>(null);
    const id = useId();

    const [buttonProps, { tagName: Component }] = useButtonProps({
      disabled,
      href,
      onClick: onClickBtn,
      ...props,
    });

    const buttonClass = getModuleClassName(customClassName, {
      variant,
      disabled,
      active,
      size,
      prefix: 'button',
    });

    return (
      <>
        <Component id={id} {...buttonProps} {...props} className={buttonClass}>
          {children}
        </Component>
      </>
    );
  }
);

ButtonCommon.displayName = 'ButtonCommon';

export default ButtonCommon;
