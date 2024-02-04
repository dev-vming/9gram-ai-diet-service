import { BaseEntity, Column, Entity, PrimaryColumn } from "typeorm";

@Entity()
export class FoodInfoNara extends BaseEntity {
  @PrimaryColumn()
  foodInfoId: string;

  @Column()
  foodName: string;

  @Column({ nullable: true })
  brand: string;

  @Column({ nullable: true })
  totalCapacity: number;

  @Column({ nullable: true })
  totalCapacityUnit: string;

  @Column({ nullable: true })
  calories: number;

  @Column({ nullable: true })
  carbohydrates: number;

  @Column({ nullable: true })
  proteins: number;

  @Column({ nullable: true })
  fats: number;

  @Column({ nullable: true })
  researchYear: string;
}
