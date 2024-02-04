import { ReactNode } from 'react';
import { createPortal } from 'react-dom';

const ToastPortal = ({ children }: { children: ReactNode | ReactNode[] }) => {
  return createPortal(children, document.getElementById('notice')!);
};

export default ToastPortal;
