import { ReactElement, useCallback } from 'react';
import { useDropdownContext } from './Dropdown';

type DropdownItemProps = {
  children?: string | ReactElement[];
  id: number;
  value?: string;
};

const DropdownItem = ({ children, id, value }: DropdownItemProps) => {
  const {
    isOpen,
    setIsopen,
    highlightedindex,
    setHighlightedIndex,
    setBtnText,
  } = useDropdownContext();
  const setText = useCallback(() => {
    if (value) {
      setBtnText(value);
    }
    if (typeof children === 'string') {
      setBtnText(children);
    }
    setHighlightedIndex(id);
    return setIsopen((prev) => !prev);
  }, [children, id, value]);

  return (
    <>
      {isOpen && typeof children === 'string' ? (
        <option onClick={setText}>{children}</option>
      ) : (
        { children }
      )}
    </>
  );
};

export default DropdownItem;
