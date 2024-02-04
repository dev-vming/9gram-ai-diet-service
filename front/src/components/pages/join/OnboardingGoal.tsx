import React, { useEffect, useRef, useState } from 'react';
import ButtonCommon from '../../UI/ButtonCommon';
import './Onboarding.css';
import { OnboardingProps } from './OnboardingGender';

const OnboardingGoal: React.FC<OnboardingProps> = ({
  userData,
  onClickOnboarding,
}) => {
  const onClickGoal = (value: number) => {
    if (userData.dietGoal === value.toString()) {
      onClickOnboarding({ ['dietGoal']: '' });
    } else {
      onClickOnboarding({ ['dietGoal']: value });
    }
  };
  return (
    <div className='onboarding-container'>
      <h1 className='b-medium'>당신의 목표는 무엇인가요?</h1>
      <div style={{ marginTop: '30px' }}>
        <ButtonCommon
          variant={Number(userData.dietGoal) === 1 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClickGoal(1)}
        >
          체중감량
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.dietGoal) === 2 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClickGoal(2)}
        >
          체중유지
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.dietGoal) === 3 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClickGoal(3)}
        >
          체중증량
        </ButtonCommon>
      </div>
      <div style={{ marginTop: '15px' }}>
        <ButtonCommon
          variant={Number(userData.dietGoal) === 4 ? 'active' : 'default'}
          size='large'
          onClickBtn={() => onClickGoal(4)}
        >
          근육증량
        </ButtonCommon>
      </div>
    </div>
  );
};

export default OnboardingGoal;
