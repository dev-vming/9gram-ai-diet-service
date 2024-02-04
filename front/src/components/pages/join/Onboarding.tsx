import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import ButtonCommon from '../../UI/ButtonCommon';
import OnboardingGender from './OnboardingGender';
import OnboardingBirth from './OnboardingBirth';
import OnboardingHeight from './OnboardingHeight';
import OnboardingWeight from './OnboardingWeight';
import OnboardingGoal from './OnboardingGoal';
import OnboardingActivity from './OnboardingActivity';
import useCachingApi from '@hooks/useCachingApi';
import { checkValuesNullOrEmpty } from '@utils/checkValuesNullOrEmpty';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '@components/store';
import { UserInfo } from '@components/store/userLoginRouter';
import RenderProgressBar from './RenderProgressBar';
import getNutritionStandard from '@utils/getNutritionStandard';
import {
  adjustCaloriesByGoal,
  calBMR,
  calBMRCalories,
} from '../my-page/calUserData';
import { loginUser } from '@components/store/userLoginRouter';

export interface userPutDataType {
  dietGoal?: string;
  activityAmount?: string;
  height?: number;
  weight?: number;
  gender?: string;
  age?: number;
  birthDay?: string;
}

interface OnBoardingResult {
  data: any;
  status: number;
}

const initialUserInfo = {
  activityAmount: undefined,
  age: 0,
  birthDay: '',
  dietGoal: '',
  gender: undefined,
  height: undefined,
  targetCalories: 0,
  username: undefined,
  weight: undefined,
};

const Onboarding = () => {
  const navigate = useNavigate();
  const { step } = useParams();
  const [curStep, setCurStep] = useState(Number(step) || 1);
  const [userData, setUserData] = useState<any>(initialUserInfo);
  const [navigateTrigger, setNavigateTrigger] = useState(false);

  const dispatch = useDispatch();

  const returnedUserData = useSelector(
    (state: RootState) => state.user.userInfo
  );
  const { loading, trigger } = useCachingApi<any>({
    method: 'put',
    path: '/user',
  });
  const changeTypedUserData: UserInfo = Object.assign(
    {},
    {
      activityAmount:
        returnedUserData.activityAmount &&
        returnedUserData.activityAmount.toString(),
      dietGoal: returnedUserData.dietGoal,
      height: returnedUserData.height,
      gender: returnedUserData.gender,
      birthDay: returnedUserData.birthDay,
      weight: returnedUserData.weight,
    }
  );
  useEffect(() => {
    navigateTrigger && navigate('/home');
  }, [navigateTrigger]);

  //fetched된 userData중 일부만 가져옴
  useEffect(() => {
    setUserData(changeTypedUserData);
  }, [returnedUserData]);

  const onClickTrigger = () => {
    trigger(
      { ...userData },
      {
        onSuccess: (data) => {
          if (data.data.targetCalories && data.status === 200) {
            dispatch(loginUser(userData));
            setNavigateTrigger(true);
          }
        },
      }
    );
  };

  //목표칼로리, 영양성분 userData에 업데이트
  const updateCalculatedData = (userData: any) => {
    //goalCalrories 계산
    const bmr = calBMR({ data: userData });
    const bmrCalories = calBMRCalories({
      bmr,
      data: userData,
    });
    const goalCalories = Math.round(
      adjustCaloriesByGoal({ data: userData, bmrCalories })
    );
    //영양성분 계산
    const { carbohydrates, proteins, fats, dietaryFiber } =
      getNutritionStandard(userData);

    //userData에 세팅
    setUserData((prev: any) => ({
      ...prev,
      ['targetCalories']: goalCalories,
      ['recommendIntake']: {
        carbohydrates,
        proteins,
        fats,
        dietaryFiber,
      },
    }));
  };

  const onNextClick = async () => {
    if (curStep === 6) {
      //step마지막일때 객체가 비어있는지 확인 후
      //비어있는 곳으로 이동
      if (checkValuesNullOrEmpty(userData)) {
        setCurStep(checkValuesNullOrEmpty(userData) as number);
        return navigate(`/onboarding/${checkValuesNullOrEmpty(userData)}`);
      } else {
        updateCalculatedData(userData);
      }
    } else {
      //step 1~5
      setCurStep((prev) => prev + 1);
      navigate(`/onboarding/${curStep + 1}`);
    }
  };

  const onClickOnboarding = (onboardingInfo: userPutDataType) => {
    setUserData((prev: any) => ({
      ...prev,
      ...onboardingInfo,
    }));
  };

  //업데이트 영양성분과 목표 칼로리 모두 있을떄 감지해서 PUT요청
  useEffect(() => {
    if (
      userData.activityAmount &&
      userData.age &&
      userData.birthDay &&
      userData.dietGoal &&
      userData.gender &&
      userData.height &&
      userData.recommendIntake?.carbohydrates &&
      userData.recommendIntake?.dietaryFiber &&
      userData.recommendIntake?.fats &&
      userData.recommendIntake?.proteins
    ) {
      onClickTrigger();
    }
  }, [
    userData.activityAmount &&
      userData.age &&
      userData.birthDay &&
      userData.gender &&
      userData.height &&
      userData.recommendIntake?.carbohydrates &&
      userData.recommendIntake?.dietaryFiber &&
      userData.recommendIntake?.fats &&
      userData.recommendIntake?.proteins,
  ]);

  const isNextButtonDisabled = () => {
    switch (curStep) {
      case 1:
        return userData.gender === '';
      case 2:
        return userData.birthDay === '';
      case 3:
        return userData.height?.toString() === '';
      case 4:
        return userData.weight?.toString() === '';
      case 5:
        return userData.dietGoal === '';
      case 6:
        return userData.activityAmount === '';
      default:
        return true;
    }
  };

  return (
    <div
      style={{
        minHeight: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
      }}
    >
      <div>
        <div className='progress-bar' style={{ marginBottom: '50px' }}>
          {RenderProgressBar(curStep)}
        </div>
        {curStep === 1 && (
          <OnboardingGender
            userData={userData}
            onClickOnboarding={onClickOnboarding}
          />
        )}
        {curStep === 2 && (
          <OnboardingBirth
            userData={userData}
            onClickOnboarding={onClickOnboarding}
          />
        )}
        {curStep === 3 && (
          <OnboardingHeight
            userData={userData}
            onClickOnboarding={onClickOnboarding}
          />
        )}
        {curStep === 4 && (
          <OnboardingWeight
            userData={userData}
            onClickOnboarding={onClickOnboarding}
          />
        )}
        {curStep === 5 && (
          <OnboardingGoal
            userData={userData}
            onClickOnboarding={onClickOnboarding}
          />
        )}
        {curStep === 6 && (
          <OnboardingActivity
            userData={userData}
            onClickOnboarding={onClickOnboarding}
          />
        )}
      </div>
      <div className='button-container'>
        <ButtonCommon
          variant='default-active'
          size='big'
          onClickBtn={onNextClick}
          disabled={isNextButtonDisabled()}
        >
          다음
        </ButtonCommon>
      </div>
    </div>
  );
};

export default Onboarding;
