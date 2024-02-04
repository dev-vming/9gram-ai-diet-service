import React, { useEffect } from 'react';
import ButtonCommon from '../../UI/ButtonCommon';
import './Onboarding.css';
import { OnboardingProps } from './OnboardingGender';

const OnboardingActivity: React.FC<OnboardingProps> = ({
  userData,
  onClickOnboarding,
}) => {
  const onClickActivity = (value: number) => {
    if (userData.activityAmount === value.toString()) {
      onClickOnboarding({ ['activityAmount']: '' });
    } else {
      onClickOnboarding({ ['activityAmount']: value.toString() });
    }
  };
  return (
    <div className='onboarding-container'>
      <h1 className='b-medium'>당신의 평소 활동량은 어떤가요?</h1>
      <div style={{ marginTop: '30px' }}>
        <ButtonCommon
          variant={
            Number(Number(userData.activityAmount)) === 1 ? 'active' : 'default'
          }
          size='large'
          onClickBtn={() => onClickActivity(1)}
        >
          매우 활동적
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.activityAmount) === 2 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClickActivity(2)}
        >
          활동적
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.activityAmount) === 3 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClickActivity(3)}
        >
          저활동적
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.activityAmount) === 4 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClickActivity(4)}
        >
          비활동적
        </ButtonCommon>
      </div>
    </div>
  );
};

export default OnboardingActivity;
