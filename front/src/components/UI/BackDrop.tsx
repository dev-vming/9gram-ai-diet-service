import { FC, PropsWithChildren } from 'react';
import classes from './backDrop.module.css';

type Props = {
  onClick?: React.MouseEventHandler<HTMLDivElement>;
};

const BackDrop: FC<PropsWithChildren<Props>> = ({ children, onClick }) => {
  return (
    <div className={classes.backdrop} onClick={onClick}>
      {children}
    </div>
  );
};

export default BackDrop;
