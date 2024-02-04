import { Column, Entity, PrimaryColumn } from "typeorm";

@Entity()
export class FoodInfo {
  @PrimaryColumn()
  foodInfoId: string;

  @Column()
  foodName: string;

  @Column({ nullable: true })
  category: string;

  @Column({ nullable: true })
  mainCode: string;

  @Column({ nullable: true })
  mainFoodName: string;

  @Column({ nullable: true })
  brand: string;

  @Column({ nullable: true })
  standardCapacity: number;

  @Column({ nullable: true })
  totalCapacity: number;

  @Column({ nullable: true })
  calories: number;

  @Column({ nullable: true })
  carbohydrates: number;

  @Column({ nullable: true })
  proteins: number;

  @Column({ nullable: true })
  fats: number;

  @Column({ nullable: true })
  dietaryFiber: number;

  @Column({ nullable: true })
  createdDate: Date;

  @Column({ nullable: true })
  updatedDate: Date;
}
