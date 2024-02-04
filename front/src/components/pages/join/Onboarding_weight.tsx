import { useEffect, useRef, useState } from "react";
import './Onboarding.css';
import NumericPad from "./NumericPad";

const Onboarding_weight = () => {
  const inputWeight = useRef(null);

  return (
    <div>
      <h1 className="b-medium">당신의 몸무게는 얼마인가요?</h1>
      <div style={{ marginTop: '130px', textAlign: 'center' }}>
        <input
          type="text"
          ref={inputWeight}
          placeholder="100"
          className="b-super"
          style={{ width: '150px', textAlign: 'center', display: 'inline-block', border: 'none' }}
        />
        <span
          style={{ marginLeft: '5px', verticalAlign: 'top' }}
          className="b-super"
        >kg</span>
      </div>
      <div style={{ marginTop: '50px' }}>
        <NumericPad inputRef={inputWeight} />
      </div>
    </div>
  );
};

export default Onboarding_weight;