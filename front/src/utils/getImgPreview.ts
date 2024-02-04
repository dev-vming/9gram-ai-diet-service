const getImgPreview = (
  previewImgFile: Blob,
  setProfileImage: React.Dispatch<React.SetStateAction<string | undefined>>,
  callback: (file: File | string) => void
) => {
  const reader = new FileReader();

  reader.onload = (event) => {
    setProfileImage(event.target?.result as string);
    if (previewImgFile instanceof File) {
      callback(previewImgFile);
    }
  };
  reader.readAsDataURL(previewImgFile);
};

export default getImgPreview;
