import { GetClassNameType } from 'typings/commontypes';

export const getClassNamesArr = (
  classNames: string | undefined,
  arr: any[]
): string[] | string | undefined => {
  let cls: string[] = [];
  if (arr.length > 0) {
    cls = cls?.concat(
      arr.reduce((acc: string[], [key, val]: GetClassNameType[]) => {
        //val이 유효할 경우에만 key-value
        val && acc.push(`${key.toString()}-${val?.toString()}`);
        return acc;
      }, [])
    );
    classNames && cls?.unshift(classNames.toString());
  }
  return cls;
};
