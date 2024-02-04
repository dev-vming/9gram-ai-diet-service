import { ReactNode } from 'react';
import classes from './toast.module.css';

const ToastText = ({
  children,
  style,
  toastPosCls,
}: {
  children: ReactNode | string;
  style?: {};
  toastPosCls?: 'top' | 'middle' | 'bottom';
}) => {
  return (
    <div
      className={`${classes.wrapper} ${classes[`${toastPosCls}`]}`}
      style={style}
    >
      {children}
    </div>
  );
};

export default ToastText;
