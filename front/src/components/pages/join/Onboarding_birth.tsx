import { useEffect, useRef, useState } from "react";
// import InputNumber from "../../UI/InputNumber";
import InputCommon from '@components/UI/InputCommon';
import './Onboarding.css';

const Onboarding_birth = () => {
    const [year, setYear] = useState('');
    const [month, setMonth] = useState('');
    // const [value, setValue] = useState('');
    const [date, setDate] = useState('');
    // const [selectedBirth, setSelectedBirth] = useState<string | null>(null);

    const handleYearChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setYear(e.target.value);
    };
    const handleMonthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setMonth(e.target.value);
    };
    // const valueChangeFn: React.ChangeEventHandler<HTMLInputElement> = (value) => {
    //     value && setValue(Number(value));
    // };
    const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setDate(e.target.value);
    };

    return (
        <div>
            <h1 className="b-medium">생년월일을 입력해주세요</h1>
            <div style={{ marginTop: '30px' }}>
                <InputCommon
                    variant="default"
                    value={year}
                    placeholder='년'
                    onChange={handleYearChange}
                />
            </div>
            <div style={{ display: "flex", gap: '10px', width: "350px", margin: '15px auto' }}>
                {/* <InputNumber
                    value={value}
                    maximumValue={12}
                    placeholder='월'
                    onValueChange={valueChangeFn}
                /> */}
                <InputCommon style={{ width: "170px" }}
                    variant="default"
                    value={month}
                    placeholder='월'
                    onChange={handleMonthChange}
                />
                <InputCommon style={{ width: "170px" }}
                    variant="default"
                    value={date}
                    placeholder='일'
                    onChange={handleDateChange}
                />
            </div>
        </div>
    );
};

export default Onboarding_birth;