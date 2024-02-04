import { useCallback, useRef, useState } from 'react';

interface UseControlledArgs<T = any> {
  controlled?: T;
  unControlled: T;
}

type UseControlledReturn<T = any> = [
  T,
  (value: T | React.Dispatch<React.SetStateAction<T>>) => void,
];

//매개변수 : {controlledValue, defaultValue}
export const useControlled = <T>(
  args: UseControlledArgs<T>
): UseControlledReturn => {
  const { controlled: controlledValue, unControlled } = args;

  //controlledValue가 넘어오면 isControlled
  //dom이 리렌더링되어도 이 값이 남아있기 위해 useRef로 사용 저장
  const { current: isControlled } = useRef(controlledValue !== undefined);

  const [state, setState] = useState<T>(unControlled);

  //isControlled면 controlledValue, 아니면 defaultValue를
  const value = isControlled ? controlledValue : state;
  const setValue: React.Dispatch<React.SetStateAction<T>> = useCallback(
    (newState) => {
      !isControlled && setState(newState);
    },
    [isControlled]
  );

  return [value, setValue];
};
//value와 setValue를 내놓음
//value는 controlledValue가 오면 그것을, 아니면 defaultValue를 내놓음
//setValue는 controlled상태가 아닐때 state를 새로운 값을 받아서 변경해줌
