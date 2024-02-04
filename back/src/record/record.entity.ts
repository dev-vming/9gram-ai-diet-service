import { BaseEntity, Column, Entity, PrimaryGeneratedColumn } from "typeorm";

export enum MealType {
  "아침" = 1,
  "점심" = 2,
  "저녁" = 3,
  "간식" = 4
}

export class Food {
  foodName: string
  counts: number
  XYCoordinate: number[]
}

@Entity()
export class Record extends BaseEntity {
  @PrimaryGeneratedColumn('uuid')
  recordId: string;

  @Column({ type: "varchar", length: 50, nullable: false })
  userId: string;

  @Column({
    type: "enum",
    enum: MealType,
    nullable: false
  })
  mealType: MealType;

  @Column({ type: "varchar", nullable: false })
  foodInfoId: string;

  @Column({ type: "varchar", nullable: false })
  recordFoodName : string;

  @Column({ type: "float", nullable: false })
  foodCounts: number;

  @Column({ type: "int", nullable: true })
  imageId: number;

  @Column({ type: "float", nullable: true })
  carbohydrates: number;

  @Column({ type: "float", nullable: true })
  proteins: number;

  @Column({ type: "float", nullable: true })
  fats: number;

  @Column({ type: "float", nullable: true })
  dietaryFiber: number;

  @Column({ type: "float", nullable: true })
  totalCalories: number;

  @Column({ type: "date", nullable: true })
  firstRecordDate: Date;

  @Column({ type: "timestamp with time zone", nullable: true })
  updatedDate: Date;
}