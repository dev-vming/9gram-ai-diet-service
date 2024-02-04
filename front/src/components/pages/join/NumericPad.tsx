import React from 'react';

interface NumericPadProps {
  setInputValue: React.Dispatch<React.SetStateAction<string>>;
}

const NumericPad: React.FC<NumericPadProps> = ({ setInputValue }) => {
  const handleNumericClick = (value: string | number) => {
    if (value === 'DEL') {
      setInputValue((prev) => prev.slice(0, -1));
    } else {
      setInputValue((prev) => prev + value);
    }
  };

  const numericButtons = [1, 2, 3, 4, 5, 6, 7, 8, 9, '.', 0, 'DEL'];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '10px',
        background: '#f4f4f4',
        padding: '20px 0',
      }}
    >
      {numericButtons.map((button, index) => (
        <div
          key={index}
          style={{
            padding: '10px',
            borderRadius: '0px',
            cursor: 'pointer',
            fontSize: '18px',
            textAlign: 'center',
            transition: 'color 0.3s',
          }}
          onClick={() => handleNumericClick(button)}
          onMouseOver={(e) => {
            e.currentTarget.style.color = '#346DFF';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.color = '';
          }}
        >
          {button}
        </div>
      ))}
    </div>
  );
};

export default NumericPad;
