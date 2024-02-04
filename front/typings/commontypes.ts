import { ComponentPropsWithRef, ReactElement } from 'react';

//COMMON

export type CommonSizeType =
  | 'large'
  | 'big'
  | 'medium'
  | 'med-small'
  | 'small'
  | 'ssmall'
  | 'tiny';

export type CommonFontSizeType =
  | 'b-super'
  | 'b-large'
  | 'b-big'
  | 'b-medium'
  | 'b-regular'
  | 'b-small'
  | 'b-tiny'
  | 'r-super'
  | 'r-large'
  | 'r-big'
  | 'r-medium'
  | 'r-regular'
  | 'r-small'
  | 'r-tiny'
  | 's-large'
  | 's-big'
  | 's-medium'
  | 's-tiny';

//BUTTON

export type ButtonVariant =
  | 'default'
  | 'active'
  | 'default-active'
  | 'validated'
  | 'error'
  | 'disabled';

export interface ButtonPropsType extends ComponentPropsWithRef<'button'> {
  children?: string | ReactElement | number | string[];
  customClassName?: string;
  onClickBtn?: (
    event: React.MouseEvent<HTMLButtonElement, MouseEvent>
  ) => void | undefined | any;
  href?: string;
  variant?: ButtonVariant;
  size?: CommonSizeType;
  disabled?: boolean;
  active?: boolean;
  fontSize?: CommonFontSizeType;
}

export interface GetClassNameType {
  variant?: ButtonVariant;
  disabled?: boolean;
  active?: boolean;
  size?: CommonSizeType;
  prefix?: string;
}

// INPUT

export type InputVariant =
  | 'default'
  | 'active'
  | 'default-active'
  | 'validated'
  | 'error';
