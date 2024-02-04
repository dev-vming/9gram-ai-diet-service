import { useEffect } from 'react';
import { useDropdownContext } from './Dropdown';

type DropdownButtonProps = {
  children: string;
};

const DropdownButton = ({ children }: DropdownButtonProps) => {
  const { setBtnText } = useDropdownContext();

  useEffect(() => {
    setBtnText(children);
  }, []);

  return <></>;
};

export default DropdownButton;
