// import { User } from "src/user/user.entity";
import { Image } from "src/image/entities/image.entity";
import { MealType } from "src/record/record.enum";
import { User } from "src/user/entities/user.entity";
import {
  Column,
  Entity,
  JoinColumn,
  ManyToOne,
  OneToOne,
  PrimaryGeneratedColumn,
} from "typeorm";

@Entity()
export class CumulativeRecord {
  @PrimaryGeneratedColumn()
  cumulativeRecordId: string;

  @Column({ type: "int", array: true })
  // @JoinColumn()
  // @OneToOne(() => Image, (image) => image.imageId, { cascade: true })
  imageId: number;

  @Column()
  userId: string;

  @Column({ type: "enum", name: "meal_type", enum: MealType, nullable: true })
  mealType: MealType;

  @Column({ type: "float", default: 0 })
  mealTotalCalories: number;

  @Column()
  date: Date;

  @Column({ type: "float", default: 0 })
  carbohydrates: number;

  @Column({ type: "float", default: 0 })
  proteins: number;

  @Column({ type: "float", default: 0 })
  fats: number;

  @Column({ type: "float", default: 0 })
  dietaryFiber: number;
}
