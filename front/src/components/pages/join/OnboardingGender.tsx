import React, { useEffect, useState } from 'react';
import ButtonCommon from '@components/UI/ButtonCommon';
import './Onboarding.css';
import { UserInfo } from '@components/store/userLoginRouter';

export interface OnboardingProps {
  userData: UserInfo;
  onClickOnboarding: (onboardingInfo: object) => void;
}

const OnboardingGender: React.FC<OnboardingProps> = ({
  userData,
  onClickOnboarding,
}) => {
  const onClick = (gender: number) => {
    if (userData.gender === gender.toString()) {
      onClickOnboarding({ ['gender']: '' });
    } else {
      onClickOnboarding({ ['gender']: gender.toString() });
    }
  };

  return (
    <div className='onboarding-container'>
      <h1 className='b-medium'>당신의 성별은 무엇인가요?</h1>
      <div style={{ marginTop: '30px' }}>
        <ButtonCommon
          variant={Number(userData.gender) === 2 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClick(2)}
        >
          여성
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.gender) === 1 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClick(1)}
        >
          남성
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.gender) === 3 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClick(3)}
        >
          기타
        </ButtonCommon>
      </div>
    </div>
  );
};

export default OnboardingGender;
