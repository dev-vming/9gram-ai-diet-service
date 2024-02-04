import { useEffect, useState, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import style from './mypageedit.module.css';
import MyPageDropdown from './MyPageDropdwon';
import getImgPreview from '@utils/getImgPreview';
import getNutritionStandard from '@utils/getNutritionStandard';
import { mapGoaltoMsg, mapActivitytoMsg, findKeyByValue } from './mapMsg';
import ButtonCommon from '@components/UI/ButtonCommon';
import { calBMR, calBMRCalories, adjustCaloriesByGoal } from './calUserData';
import { loginUser } from '@components/store/userLoginRouter';
import { UserData } from './MypageTypes';
import useApi, { TriggerType } from '@hooks/useApi';
import usePresignedUrl from '@hooks/usePresignedUrl';
import useS3ImgUpload from '@hooks/useS3ImgUpload';

const goalTypes = ['근육증량', '체중감량', '체중유지', '체중증량'];
const activityType = ['비활동적', '약간 활동적', '활동적', '매우 활동적'];

const MyPageEdit = () => {
  const location = useLocation();
  const dispatch = useDispatch();
  const {
    trigger,
  }: {
    trigger: TriggerType;
  } = useApi({
    method: 'put',
  });

  const { userData, goalMsg, activityMsg } = location.state;
  const [data, setData] = useState(userData);

  const [previewImage, setPreviewImage] = useState<string | undefined>(
    data.profileImage
  );
  const [file, setFile] = useState<File | null>(userData.profileImage);
  const [fileChanged, setFileChanged] = useState(false);
  const [bmr, setBmr] = useState(calBMR({ data }));
  const [bmrCalories, setBmrCalories] = useState(calBMRCalories({ bmr, data }));
  const [goalCalories, setGoalCalories] = useState(
    Math.round(adjustCaloriesByGoal({ data, bmrCalories }))
  );
  const [isEditingData, setIsEditingData] = useState(false);
  const [prevWeight, setPrevWeight] = useState(data.weight);
  const [prevHeight, setPrevHeight] = useState(data.height);
  const [selectedGoal, setSelectedGoal] = useState(goalMsg);
  const [selectedActity, setSelectedActity] = useState(activityMsg);
  const [isActivityDropdownVisible, setActivityDropdownVisible] =
    useState(false);
  const [isGoalDropdownVisible, setGoalDropdownVisible] = useState(false);
  const [updated, setUpdated] = useState(false);

  const navigate = useNavigate();

  const { getPresignedUrl, presignedUrl, error, loading } = usePresignedUrl({
    fileName: file?.name,
    path: `image/presigned-url/profile/${file?.name}`,
  });

  useEffect(() => {
    if (file) {
      getPresignedUrl({
        fileName: file.name,
        path: `image/presigned-url/profile/${file.name}`,
      });
    }
  }, [getPresignedUrl, file]);

  const { uploadToS3 } = useS3ImgUpload();

  const imgInputRef = useRef<HTMLInputElement>(null);
  const handleImageClick = () => {
    imgInputRef.current?.click();
  };

  const handleImageSelect = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const previewImgFile = event.target.files?.[0] || null;

    if (previewImgFile) {
      getImgPreview(previewImgFile, setPreviewImage, (file) => {
        if (file instanceof File) {
          setFile(file);
          setFileChanged(true);
        }
      });
    } else {
      setPreviewImage(previewImage);
    }
  };

  const toggleGoalDropdown = () => {
    setGoalDropdownVisible(!isGoalDropdownVisible);
    setActivityDropdownVisible(false);
  };

  const toggleActivityDropdown = () => {
    setActivityDropdownVisible(!isActivityDropdownVisible);
    setGoalDropdownVisible(false);
  };

  const updateDataAndCalories = (updatedData: UserData) => {
    const updatedBmr = calBMR({ data: updatedData });
    const updatedBmrCalories = calBMRCalories({
      bmr: updatedBmr,
      data: updatedData,
    });
    const updatedGoalCalories = Math.round(
      adjustCaloriesByGoal({
        data: updatedData,
        bmrCalories: updatedBmrCalories,
      })
    );

    setData(updatedData);
    setBmr(updatedBmr);
    setBmrCalories(updatedBmrCalories);
    setGoalCalories(updatedGoalCalories);
  };

  const handleSelect = (type: 'goal' | 'activity', value: string) => {
    const newGoalValue = findKeyByValue(mapGoaltoMsg, value);
    if (type === 'goal' && newGoalValue) {
      const newGoal = newGoalValue;
      if (['1', '2', '3', '4'].includes(newGoal)) {
        const updatedData = {
          ...data,
          dietGoal: newGoal,
        };
        updateDataAndCalories(updatedData);
        setSelectedGoal(value);
      }
      setGoalDropdownVisible(false);
    } else if (type === 'activity') {
      const newActivityValue = findKeyByValue(mapActivitytoMsg, value);
      if (newActivityValue) {
        const newActivity = newActivityValue;
        if (['1', '2', '3', '4'].includes(newActivity)) {
          const updatedData = {
            ...data,
            activityAmount: newActivity,
          };
          updateDataAndCalories(updatedData);
          setSelectedActity(value);
        }
      }
      setActivityDropdownVisible(false);
    }
  };

  const editHeightAndWeight = () => {
    setIsEditingData(!isEditingData);
    setPrevHeight(data.height);
    setPrevWeight(data.weight);
  };

  const updateProfileData = async () => {
    try {
      const updatedData = {
        ...data,
        dietGoal: Number(findKeyByValue(mapGoaltoMsg, selectedGoal)),
        activityAmount: Number(
          findKeyByValue(mapActivitytoMsg, selectedActity)
        ),
        height: Number(prevHeight),
        weight: Number(prevWeight),
        profileImage: file,
      };
      setData(updatedData);
      updateDataAndCalories(updatedData);
    } catch (error) {
      console.log('profile data updating error', error);
    }
  };

  const uploadProfileImage = async () => {
    try {
      if (presignedUrl.data && file) {
        const uploadUrl = await uploadToS3({
          presignedUrl: presignedUrl.data,
          file,
        });
        if (uploadUrl) {
          const uploadedImageUrl = presignedUrl.data.split('?')[0];
          return uploadedImageUrl;
        }
      }
    } catch (error) {}
  };

  const saveAndNavigate = async () => {
    let uploadedImageUrl;
    let updatedNutrients = getNutritionStandard(data);

    if (!(data.dietGoal && data.targetCalories && data.gender)) {
      const updatedNutrients = {
        carbohydrates: 0,
        dietaryFiber: 0,
        proteins: 0,
        fats: 0,
      };
    }

    if (fileChanged) {
      uploadedImageUrl = await uploadProfileImage();
    }

    const updatedData = {
      ...data,
      dietGoal: Number(findKeyByValue(mapGoaltoMsg, selectedGoal)),
      activityAmount: Number(findKeyByValue(mapActivitytoMsg, selectedActity)),
      height: Number(prevHeight),
      weight: Number(prevWeight),
      recommendIntake: updatedNutrients,
      targetCalories: goalCalories,
      profileImage: uploadedImageUrl || file,
    };
    const { username, ...dataToSend } = updatedData;

    updateDataAndCalories(updatedData);
    dispatch(loginUser(updatedData));

    await trigger({
      applyResult: true,
      isShowBoundary: false,
      data: dataToSend,
      path: 'user',
    });

    setUpdated(true);
  };

  useEffect(() => {
    if (updated) {
      navigate('/my-page');
    }
  }, [updated, navigate]);

  return (
    <>
      <div className={style.userProfileArea}>
        <div className={style.userProfileContainer}>
          <input
            type='file'
            accept='image/*'
            style={{ display: 'none' }}
            ref={imgInputRef}
            onChange={handleImageSelect}
          />
          {previewImage ? (
            <>
              <img
                className={style.userProfile}
                src={previewImage}
                alt='사용자 프로필'
                onClick={handleImageClick}
              />
              <span className={style.imgButton}>+</span>
            </>
          ) : (
            <>
              <div
                className={style.defaultProfile}
                onClick={handleImageClick}
              ></div>
              <span className={style.imgButton}>+</span>
            </>
          )}

          <div className={style.userName}>{data.username}</div>
        </div>
      </div>

      <div className={style.goalInfoArea}>
        <div className={style.goalInfoTitle}>
          <div className={style.infoTitle}>목표</div>
          <MyPageDropdown
            items={goalTypes}
            selectedItem={selectedGoal ? `${selectedGoal}` : '목표 설정'}
            onSelectItem={(value) => handleSelect('goal', value)}
            toggleDropdown={toggleGoalDropdown}
            isDropdownVisible={isGoalDropdownVisible}
          />
        </div>
        <div className={style.goaltInfo}>
          <div className={style.goalTitle}>목표 칼로리</div>
          <div className={style.goalDetail}>
            {goalCalories ? `${goalCalories}` : '활동량 설정'}kcal
          </div>
        </div>
      </div>

      <div className={style.infoArea}>
        <div className={style.infoTitle}>신체 데이터</div>
        <div className={style.infoContent}>
          {isEditingData ? (
            <>
              <input
                type='number'
                className={style.inputStyle}
                value={prevHeight}
                onChange={(e) => setPrevHeight(Number(e.target.value))}
                onBlur={updateProfileData}
              />
              <span style={{ color: '#346dff' }}> cm </span>
              <span style={{ color: 'black' }}> / </span>
              <input
                type='number'
                className={style.inputStyle}
                value={prevWeight}
                onChange={(e) => setPrevWeight(Number(e.target.value))}
                onBlur={updateProfileData}
              />
              <span style={{ color: '#346dff' }}> kg </span>
            </>
          ) : (
            <>
              <div
                onClick={editHeightAndWeight}
                className={style.userDataDetail}
              >
                {data.height} cm <span style={{ color: 'black' }}> / </span>
                {data.weight} kg
              </div>
            </>
          )}
        </div>
      </div>

      <div className={style.activityAccountArea}>
        <div className={style.infoTitle}>활동량</div>
        <MyPageDropdown
          items={activityType}
          selectedItem={selectedActity}
          onSelectItem={(value) => handleSelect('activity', value)}
          toggleDropdown={toggleActivityDropdown}
          isDropdownVisible={isActivityDropdownVisible}
        />
      </div>

      <div className={style.editButton}>
        <ButtonCommon
          variant='default-active'
          size='big'
          onClick={saveAndNavigate}
        >
          수정 완료
        </ButtonCommon>
      </div>
    </>
  );
};

export default MyPageEdit;
