export interface MyPageDropdownProps {
  items: string[];
  selectedItem: string;
  onSelectItem: (item: string) => void;
  toggleDropdown: () => void;
  isDropdownVisible: boolean;
  style?: React.CSSProperties;
}

interface TargetNutrients {
  carbohydrates: number;
  proteins: number;
  fats: number;
  dietaryFiber: number;
  [key: string]: number;
}

export type UserData = {
  username?: string;
  gender?: string;
  age: number;
  profileImage?: string;
  weight: number;
  height: number;
  dietGoal: string;
  targetCalories: number;
  recommendIntake: TargetNutrients;
  activityAmount: string;
};

export interface MyPageEditProps {
  data: UserData;
  goalMsg: string;
  activityMsg: string;
}
