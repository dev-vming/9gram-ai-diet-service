export type TopNavKeyType =
  | 'join'
  | 'onboardingstep'
  | 'onboardingmeal'
  | 'login'
  | 'home'
  | 'calendar'
  | 'recordymd'
  | 'recordymdmeal'
  | 'recordymdmealedit'
  | 'addphoto'
  | 'recordeidt'
  | 'addphotoymdmealsearch'
  | 'auth';

export interface TopBarPropsType {
  home?: boolean;
  title?: string;
  back?: boolean;
  qIcon?: boolean;
  icon?: boolean;
}
