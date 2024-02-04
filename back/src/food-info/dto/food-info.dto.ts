import { Exclude, Expose, Transform } from "class-transformer";
import { IsInt, IsNotEmpty } from "class-validator";

@Exclude()
export class FoodInfoDto {
  @Expose()
  @IsNotEmpty()
  foodName: string;

  @Expose()
  @IsNotEmpty()
  foodInfoId: string;

  @Expose()
  @IsNotEmpty()
  @IsInt()
  @Transform(({ value }) => value / 100)
  totalCapacity: number;

  @Expose()
  @IsNotEmpty()
  @IsInt()
  @Transform(({ value }) => value / 100)
  calories: number;

  @Expose()
  @IsInt()
  @Transform(({ value }) => value / 100)
  carbohydrates: number;

  @Expose()
  @IsInt()
  @Transform(({ value }) => value / 100)
  proteins: number;

  @Expose()
  @IsInt()
  @Transform(({ value }) => value / 100)
  fats: number;

  @Expose()
  @IsInt()
  @Transform(({ value }) => value / 100)
  dietaryFiber: number;
}
