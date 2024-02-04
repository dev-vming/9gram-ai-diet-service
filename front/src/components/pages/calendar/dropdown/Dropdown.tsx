import {
  createContext,
  useState,
  ReactNode,
  useContext,
  useEffect,
  useMemo,
  useCallback,
  Dispatch,
} from 'react';
import classes from './dropdown.module.css';

type DropdownProps = {
  children: ReactNode | ReactNode[];
  onChange?: (val: string) => void;
  className?: string;
};

// Dispatch<React.SetStateAction<number>>

export type higlightedIndexType = number | string;

interface DropdownContextType {
  isOpen: boolean;
  setIsopen: Dispatch<React.SetStateAction<boolean>>;
  highlightedindex: higlightedIndexType;
  setHighlightedIndex: Dispatch<React.SetStateAction<number>>;
  setBtnText: Dispatch<React.SetStateAction<string>>;
}
const DropdownContext = createContext<DropdownContextType | null>(null);
export const useDropdownContext = () => {
  const context = useContext(DropdownContext);
  if (!context) {
    throw new Error('Error in creating the context');
  }
  return context;
};

export const Dropdown = ({ children, onChange, className }: DropdownProps) => {
  const [isOpen, setIsopen] = useState(false);
  const [highlightedindex, setHighlightedIndex] = useState(0);
  const [btnText, setBtnText] = useState('');

  const value: DropdownContextType = useMemo(
    () => ({
      isOpen,
      setIsopen,
      highlightedindex,
      setHighlightedIndex,
      setBtnText,
    }),
    [isOpen, setIsopen, highlightedindex, setHighlightedIndex, setBtnText]
  );

  useEffect(() => {
    onChange && onChange(btnText);
  }, [btnText]);

  return (
    <DropdownContext.Provider value={value}>
      <div className={classes.container}>
        <span
          className={`${className} ${classes.value}`}
          onClick={() => {
            setIsopen((prev) => {
              console.log('handler clicked!');
              return !prev;
            });
          }}
        >
          {btnText}
        </span>
        <div
          onClick={() => {
            setIsopen((prev) => {
              console.log('handler clicked!');
              return !prev;
            });
          }}
          className={classes.caret}
        />
        {children}
      </div>
    </DropdownContext.Provider>
  );
};
