import React, { useState, useEffect } from 'react';
import style from './mergingtags.module.css';
import { TagData, MergedTagData } from './RecordTypes';

const tagPadding = 10;
const assumedTagHeight = 23;
const MaxTagWidth = 80;

export const MergingTags: React.FC<{ tagData: TagData[] }> = ({ tagData }) => {
  const [selectedFoods, setSelectedFoods] = useState<string[]>([]);
  const [selectedTagPosition, setSelectedTagPosition] = useState<number[]>([]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (selectedFoods.length > 0 && !target.closest('.tag')) {
        setSelectedFoods([]);
        setSelectedTagPosition([]);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [selectedFoods]);

  const mergeOverlappingTags = (tags: TagData[]): MergedTagData[] => {
    let mergedTags: MergedTagData[] = [];

    tags?.forEach((tag) => {
      let isMerged = false;

      for (let mergedTag of mergedTags) {
        if (
          Math.abs((mergedTag.XYCoordinate[0] - tag.XYCoordinate[0]) * 350) <
            MaxTagWidth &&
          Math.abs((mergedTag.XYCoordinate[1] - tag.XYCoordinate[1]) * 200) <
            assumedTagHeight
        ) {
          // 겹치면 태그를 합침
          mergedTag.foodNames.push(tag.foodName);
          isMerged = true;
          break;
        }
      }
      if (!isMerged) {
        // 겹치지 않으면 새로운 태그로 추가
        mergedTags.push({
          XYCoordinate: tag.XYCoordinate,
          foodNames: [tag.foodName],
        });
      }
    });

    return mergedTags;
  };

  const validTags = tagData?.filter((tag) => tag.XYCoordinate.length > 0);
  const mergedTagData = mergeOverlappingTags(validTags);

  const handleTagClick = (
    event: React.MouseEvent<HTMLElement>,
    foodNames: string[],
    XYCoordinate: number[]
  ) => {
    event.stopPropagation();
    if (foodNames.length === 1) {
      setSelectedFoods([]);
      setSelectedTagPosition([]);
    } else {
      setSelectedFoods(foodNames);
      setSelectedTagPosition(XYCoordinate);
    }
  };

  return (
    <>
      <div>
        {mergedTagData.map((tag: MergedTagData, index: number) => {
          return (
            <div className={style.mergeTagContainer} key={index}>
              <p
                style={{
                  left: `${tag.XYCoordinate[0] * 350}px`,
                  top: `${tag.XYCoordinate[1] * 200}px`,
                  maxWidth: `${MaxTagWidth}px`,
                }}
                className={`${style.tag} b-tiny`}
                onClick={(event) =>
                  handleTagClick(event, tag.foodNames, tag.XYCoordinate)
                }
              >
                {tag.foodNames.join(', ')}
              </p>
              {tag.foodNames.length > 1 && (
                <p
                  className={style.tagCount}
                  style={{
                    left: `${tag.XYCoordinate[0] * 350 + MaxTagWidth - 12}px`,
                    top: `${tag.XYCoordinate[1] * 200 - 5}px`,
                  }}
                >
                  {tag.foodNames.length}
                </p>
              )}
            </div>
          );
        })}

        {selectedFoods.length > 0 && (
          <div
            style={{
              position: 'absolute',
              left: `${selectedTagPosition[0] * 350}px`,
              top: `${selectedTagPosition[1] * 200 + 23}px`,
            }}
          >
            <div className={style.selectedTagBox}>
              {selectedFoods.map((name: string, index: number) => (
                <p
                  key={index}
                  style={{ maxWidth: `${name.length * 10 + tagPadding + 5}px` }}
                  className={`${style.selectedTag} b-tiny`}
                >
                  {name}
                </p>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
};
