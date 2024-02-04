import {
  Equal,
  LessThanOrEqual,
  MoreThanOrEqual,
  Repository,
} from "typeorm";
import { MealType, Record } from "./record.entity";
import { InjectRepository } from "@nestjs/typeorm";
import {
  BadRequestException,
  ConflictException,
  Injectable,
  NotFoundException,
} from "@nestjs/common";
import { CumulativeRecord } from "src/cumulative-record/cumulative-record.entity";
import { FoodInfo } from "src/food-info/food-info.entity";
import { Image } from "src/image/entities/image.entity";
import { SplitImage } from "src/image/entities/splitImage.entity";
import { RecordDto } from "./dtos/record.dto";
import { HealthInfo } from "src/user/entities/health-info.entity";
import { User } from "src/user/entities/user.entity";

@Injectable()
export class RecordRepository extends Repository<Record> {
  constructor(
    @InjectRepository(Record) private recordRepository: Repository<Record>,
    @InjectRepository(FoodInfo) private foodInfoRepository: Repository<FoodInfo>,
    @InjectRepository(CumulativeRecord) private cumulativeRecordRepository: Repository<CumulativeRecord>,
    @InjectRepository(Image) private imageRecordRepository: Repository<Image>,
    @InjectRepository(SplitImage) private splitImageRepository: Repository<SplitImage>,
    @InjectRepository(HealthInfo) private healthInfoRepository: Repository<HealthInfo>,
    @InjectRepository(User) private userRepository: Repository<User>
  ) {
    super(
      recordRepository.target,
      recordRepository.manager,
      recordRepository.queryRunner
    );
  }

  // 식단 조회
  async findByDate(userId: string, date: string): Promise<any> {
    const dateObj = new Date(date);
    const records = await this.recordRepository.find({
      where: {
        userId: userId,
        firstRecordDate: Equal(dateObj),
      },
    });

    const user = await this.userRepository.findOneBy({
      userId: userId
    })
  
    // 사용자의 건강 정보를 조회
    let healthInfo = await this.healthInfoRepository.findOneBy({
      healthInfoId: user.recentHealthInfoId
    })
  
    // 기본 식단 정보 객체를 준비
    let defaultMealInfo = {
      targetCalories: 0,
      recommendNutrient: {
        carbohydrates: 0,
        proteins: 0,
        fats: 0,
        dietaryFiber: 0,
      },
    };

    // healthInfo 객체가 존재하는 경우, 기본 식단 정보를 업데이트
    if (healthInfo && healthInfo.recommendIntake) {
      defaultMealInfo = {
        targetCalories: healthInfo.targetCalories || 0, // healthInfo.targetCalories가 없다면 0을 사용
        recommendNutrient: {
          carbohydrates: healthInfo.recommendIntake[0] || 0,
          proteins: healthInfo.recommendIntake[1] || 0,
          fats: healthInfo.recommendIntake[2] || 0,
          dietaryFiber: healthInfo.recommendIntake[3] || 0,
        },
      };
    }
  
    // mealType에 따라 정보를 누적할 객체를 준비
    const mealAccumulator = {};
    for (let mealType = 1; mealType <= 4; mealType++) {
      mealAccumulator[mealType] = { 
        foods: [], 
        totalCalories: 0, 
        totalNutrient: { carbohydrates: 0, proteins: 0, fats: 0, dietaryFiber: 0 }, 
        ...defaultMealInfo 
      };
    }
  
    // 조회된 레코드들을 처리
    for (const record of records) {
      let foodImage = null;
      if (record.imageId) {
        foodImage = await this.imageRecordRepository.findOne({
          where: {
            imageId: record.imageId
          }
        });
      }

      const foodInfo = await this.foodInfoRepository.findOne({
        where: {
          foodInfoId: record.foodInfoId,
        },
      });
      
      if (!foodInfo) {
        continue; // 음식 정보가 없는 경우에는 이 레코드를 건너뜀
      }
  
      const splitImageRecord = await this.splitImageRepository.findOne({
        where: {
          imageId: record.imageId,
          foodName: record.recordFoodName
        },
      });
  
      const mealType = record.mealType;
      const foodData = {
        foodInfoId: record.foodInfoId,
        recordId: record.recordId,
        foodName: record.recordFoodName,
        counts: record.foodCounts,
        XYCoordinate: splitImageRecord ? [
          Number(splitImageRecord.xCoordinate), 
          Number(splitImageRecord.yCoordinate), 
          Number(splitImageRecord.width), 
          Number(splitImageRecord.height)] : [],
      };
  
      // 각 영양소의 누적 값을 계산
      mealAccumulator[mealType].totalCalories += Math.round(record.totalCalories / 100);
      mealAccumulator[mealType].totalNutrient.carbohydrates += Math.round(record.carbohydrates / 100);
      mealAccumulator[mealType].totalNutrient.proteins += Math.round(record.proteins / 100);
      mealAccumulator[mealType].totalNutrient.fats += Math.round(record.fats / 100);
      mealAccumulator[mealType].totalNutrient.dietaryFiber += Math.round(record.dietaryFiber / 100);
      // 해당 식사 유형에 음식 데이터를 추가하면서 imgUrl을 설정
      if (!mealAccumulator[mealType].imgUrl && record.imageId) {
        const foodImage = await this.imageRecordRepository.findOne({
          where: { imageId: record.imageId }
        });
        mealAccumulator[mealType].imgUrl = foodImage ? foodImage.foodImageUrl : null;
      }
      // 해당 식사 유형에 음식 데이터를 추가
      mealAccumulator[mealType].foods.push(foodData);
    } 
  
    // 최종 정보 반환
    const response = {};
    for (const mealType in mealAccumulator) {
      if (mealAccumulator[mealType].foods.length > 0) {
        response[mealType] = mealAccumulator[mealType];
      } else {
        // foods 배열이 비어 있으면 defaultMealInfo만 포함
        response[mealType] = { ...defaultMealInfo };
      }
    }

    return response;
  }

  // 식사 기록 생성 및 수정
  async createOrUpdateRecord(userId: string, date: string, mealType: number, recordDto: RecordDto): Promise<any> {
    const recordDate = new Date(date);
    const { imgUrl, foods } = recordDto;
    console.log( "imgUrl 과연 있을까: ", imgUrl)
    // 기존 레코드 검색
    const existingRecords = await this.recordRepository.find({
      where: {
        userId,
        mealType,
        firstRecordDate: recordDate,
      },
    });
    let imageId;

    if (imgUrl) {
      const imageRecord = await this.imageRecordRepository.findOneBy({
        foodImageUrl: imgUrl,
      });
  
      if (!imageRecord) {
        // imgUrl이 존재하지만 이미지 레코드를 찾을 수 없는 경우, 새로운 레코드 생성
        const newImageRecord = this.imageRecordRepository.create({
          foodImageUrl: imgUrl
        });
  
        // 데이터베이스에 저장하고 imageId를 얻음
        const savedImageRecord = await this.imageRecordRepository.save(newImageRecord);
        imageId = savedImageRecord.imageId;
      } else {
        // imgUrl이 존재하고 이미지 레코드를 찾은 경우, 기존의 imageId 사용
        imageId = imageRecord.imageId;
      }
    } else {
      // imgUrl이 undefined인 경우, imageId를 null로 설정
      imageId = null;
    }
    
    // 기존 데이터 업데이트
    const updatedFoodIds = new Set();
    for (const food of foods) {
      // 동일한 mealType과 foodInfoId를 가진 레코드를 찾기
      const existingRecord = await this.recordRepository.findOne({
        where: {
          userId: userId,
          mealType: mealType,
          recordFoodName: food.foodName,
          firstRecordDate: Equal(recordDate)
        }
      });
      console.log("existingRecord :", existingRecord)
      const foodInfo = await this.foodInfoRepository.findOneBy({
        foodInfoId: food.foodInfoId,
      });

      if (existingRecord) {
        // 기존 레코드의 영양소 값을 업데이트
        const updatedRecord = {
          ...existingRecord,
          imageId: imageId,
          foodCounts: food.counts,
          carbohydrates: foodInfo.carbohydrates * food.counts,
          proteins: foodInfo.proteins * food.counts,
          fats: foodInfo.fats * food.counts,
          dietaryFiber: foodInfo.dietaryFiber * food.counts,
          totalCalories: foodInfo.calories * food.counts,
          updatedDate: new Date()
        };
    
        // 레코드 업데이트
        await this.recordRepository.save(updatedRecord);

      // split_image 테이블에서 해당 imageId를 가진 레코드를 찾아 정보를 업데이트
      const splitImageRecord = await this.splitImageRepository.findOneBy({
        imageId: existingRecord.imageId,
        foodName: existingRecord.recordFoodName
      });

      if (splitImageRecord) {
        // 전달받은 XY 좌표 정보로 split_image 레코드를 업데이트
        splitImageRecord.foodName = food.foodName;
        splitImageRecord.xCoordinate = food.XYCoordinate[0];
        splitImageRecord.yCoordinate = food.XYCoordinate[1];
        splitImageRecord.width = food.XYCoordinate[2];
        splitImageRecord.height = food.XYCoordinate[3];
        await this.splitImageRepository.save(splitImageRecord);
      }
      updatedFoodIds.add(food.foodInfoId);
    } else {
      // 존재하지 않으면 새로운 레코드를 생성
      await this.recordRepository.insert({
        userId: userId,
        mealType: mealType,
        foodInfoId: food.foodInfoId,
        recordFoodName: food.foodName,
        foodCounts: food.counts,
        imageId: imageId,
        carbohydrates: foodInfo.carbohydrates * food.counts,
        proteins: foodInfo.proteins * food.counts,
        fats: foodInfo.fats * food.counts,
        dietaryFiber: foodInfo.dietaryFiber * food.counts,
        totalCalories: foodInfo.calories * food.counts,
        firstRecordDate: new Date(recordDate),
        updatedDate: new Date(),
      });      
      updatedFoodIds.add(food.foodInfoId);

      const splitImageEntry = this.splitImageRepository.create({
        imageId: imageId,
        foodName: food.foodName,
        xCoordinate: food.XYCoordinate[0],
        yCoordinate: food.XYCoordinate[1],
        width: food.XYCoordinate[2],
        height: food.XYCoordinate[3],
        createdAt: new Date(),
      });

      await this.splitImageRepository.save(splitImageEntry);
    }

    // 있다가 없어진 데이터 삭제
    const recordsToDelete = existingRecords.filter(record => !updatedFoodIds.has(record.foodInfoId));
    for (const record of recordsToDelete) {
      const foodForDelete = await this.foodInfoRepository.findOne({
        where: {
          foodInfoId: record.foodInfoId
        }
      });

        await this.splitImageRepository.delete({
          imageId: record.imageId,
          foodName: record.recordFoodName
        });

        // 레코드 삭제
        await this.recordRepository.delete(record.recordId);  
      }
      // 누적 영양 정보 업데이트
      await this.updateCumulativeNutrients(userId, mealType, recordDate);    
    }
  }

  // 누적 정보 업데이트 로직
  async updateCumulativeNutrients(userId: string, mealType: number, date: Date) {
    // 누적 영양소 정보를 계산할 객체 초기화
    let cumulative = {
      carbohydrates: 0,
      proteins: 0,
      fats: 0,
      dietaryFiber: 0,
      mealTotalCalories: 0
    };

    let imageId = null;

    // 해당 사용자의 모든 식사 기록을 검색
    const records = await this.recordRepository.find({
      where: { userId, mealType, firstRecordDate: date }
    });

    // 각 기록에 대해 누적 영양소 및 칼로리 계산
    records.forEach(record => {
      cumulative.carbohydrates += Math.round(record.carbohydrates);
      cumulative.proteins += Math.round(record.proteins);
      cumulative.fats += Math.round(record.fats);
      cumulative.dietaryFiber += Math.round(record.dietaryFiber);
      cumulative.mealTotalCalories += Math.round(record.totalCalories);

      if (record.imageId) {
        imageId = record.imageId;
      }
    });

    // 해당 날짜와 식사 유형에 대한 누적 레코드를 찾음
    const cumulativeRecord = await this.cumulativeRecordRepository.findOne({
      where: { userId, mealType, date }
    });

    if (cumulativeRecord) {
      // 기존 누적 레코드 업데이트
      cumulativeRecord.carbohydrates = Math.round(cumulative.carbohydrates);
      cumulativeRecord.proteins = Math.round(cumulative.proteins);
      cumulativeRecord.fats = Math.round(cumulative.fats);
      cumulativeRecord.dietaryFiber = Math.round(cumulative.dietaryFiber);
      cumulativeRecord.mealTotalCalories = Math.round(cumulative.mealTotalCalories);   
      cumulativeRecord.imageId = imageId;   
      await this.cumulativeRecordRepository.save(cumulativeRecord);
    } else {
      // 새로운 누적 레코드 생성
      const newCumulativeRecord = this.cumulativeRecordRepository.create({
        userId,
        mealType,
        date,
        imageId,
        ...cumulative
      });
      await this.cumulativeRecordRepository.save(newCumulativeRecord);
    }
  }

  // 식단 삭제
  async deleteRecord(userId: string, date: string, mealType: number): Promise<void> {
    const dateObj = new Date(date);

    // 해당 날짜와 식사 유형에 맞는 레코드를 가져오기
    const recordsToDelete = await this.recordRepository.find({
      where: {
        userId: userId,
        firstRecordDate: Equal(dateObj),
        mealType: mealType,
      },
    });

    if (recordsToDelete.length === 0) {
      throw new NotFoundException(`해당 날짜나 식사 유형에 대한 기록이 없습니다.`);
    }

    // splitImage 레코드를 삭제
    for (const record of recordsToDelete) {
      await this.splitImageRepository.delete({
        imageId: record.imageId
      });
    }

    // Record 삭제 수행
    await this.recordRepository.remove(recordsToDelete);

    // 연관된 CumulativeRecord를 삭제
    const cumulativeRecordResult = await this.cumulativeRecordRepository.delete({
      userId: userId,
      date: Equal(dateObj),
      mealType: mealType,
    });

    if (cumulativeRecordResult.affected === 0) {
      console.log(`해당 날짜나 식사 유형에 대한 누적 기록이 없습니다.`);
    }
  }
}