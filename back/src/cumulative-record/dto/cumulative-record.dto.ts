import { IsInt, IsNotEmpty, IsUUID } from "class-validator";
import { ApiProperty } from "@nestjs/swagger";
import { Exclude, Expose, Transform } from "class-transformer";
import { MealType } from "src/record/record.enum";

@Exclude()
export class CumulativeRecordDateDto {
  @Expose()
  @ApiProperty({ description: "식사별 섭취한 총 칼로리" })
  @IsNotEmpty()
  @IsInt()
  @Transform(({ value }) => value / 100)
  totalCalories: number;

  @Expose()
  @ApiProperty({ description: "식사별 섭취한 탄수화물" })
  @IsNotEmpty()
  @Transform(({ value }) => value / 100)
  carbohydrates: number;

  @Expose()
  @ApiProperty({ description: "식사별 섭취한 단백질" })
  @IsNotEmpty()
  @Transform(({ value }) => value / 100)
  proteins: number;

  @Expose()
  @ApiProperty({ description: "식사별 섭취한 지방" })
  @IsNotEmpty()
  @Transform(({ value }) => value / 100)
  fats: number;

  @Expose()
  @ApiProperty({ description: "식사별 섭취한 식이섬유" })
  @IsNotEmpty()
  @Transform(({ value }) => value / 100)
  dietaryFiber: number;
}

@Exclude()
export class CumulativeDateMealTypeDto {
  @Expose()
  @ApiProperty({ description: "날짜" })
  @IsNotEmpty()
  date: Date;

  @Expose()
  @ApiProperty({ description: "식사 종류" })
  @IsInt()
  mealType: MealType;

  @Expose()
  @ApiProperty({ description: "하루 섭취한 총 칼로리" })
  @IsNotEmpty()
  @IsInt()
  @Transform(({ value }) => value / 100)
  mealTotalCalories: number;

  @Expose()
  @ApiProperty({ description: "하루 섭취한 총 칼로리" })
  @IsNotEmpty()
  @IsInt()
  imageId: number;
}
