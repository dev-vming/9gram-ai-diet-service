import { Entity, Column, PrimaryColumn, UpdateDateColumn, OneToOne, JoinColumn, DeleteDateColumn, Timestamp, ManyToOne } from 'typeorm';
import { IsNotEmpty, IsInt, IsString, IsDate } from 'class-validator';
import { User } from './user.entity';
import { v4 as uuidv4 } from 'uuid';
import { SaveHealthInfoDto } from '../dto/SaveHealthInfo.dto';
import { ActivityAmount, DietGoal } from '../utils/health-info.enums';
import { removeUndefined } from 'src/util/remove-undefined';

@Entity()
export class HealthInfo {

    constructor(){
        this.healthInfoId = uuidv4();
    }

    @PrimaryColumn({ type: 'uuid'})
    @IsString()
    @IsNotEmpty()
    healthInfoId: string;

    @Column({ type: 'int', nullable: true})
    @IsInt()
    weight: number;

    @Column({ type: 'int', nullable: true})
    @IsInt()
    height: number;

    @Column({ type: 'int', nullable: true})
    @IsInt()
    targetWeight: number;

    @Column({ type: 'enum',  enum: DietGoal,  nullable: true })
    dietGoal: DietGoal;

    @Column({ type: 'enum',  enum: ActivityAmount, nullable: true })
    activityAmount: ActivityAmount;

    @Column({ type: 'int', nullable: true})
    @IsInt()
    targetCalories: number;

    @Column({ type: 'int', array: true, nullable: true })
    @IsInt({ each: true })
    recommendIntake: number[];

    @Column({type: 'timestamp with time zone', default: () => 'CURRENT_TIMESTAMP'})
    @UpdateDateColumn()
    @IsDate()
    updatedDate: Timestamp;

    @Column({ type: 'timestamp with time zone', nullable: true })
    @DeleteDateColumn()
    deletedDate: Timestamp;

    // @ManyToOne((type) => User, user => user.healthInfos)
    // @JoinColumn({ name: 'user_id'})
    // user: User;

    @Column({ type: 'uuid', nullable: true })
    @IsString()
    @IsNotEmpty()
    userId: string;

    public mapHealthInfoDto(dto: SaveHealthInfoDto){
                let healthInfo = new HealthInfo();
                healthInfo.weight = dto.weight;
                healthInfo.height = dto.height;
                healthInfo.targetWeight = dto.targetWeight;
                healthInfo.dietGoal = dto.dietGoal;
                healthInfo.activityAmount = dto.activityAmount;
                healthInfo.targetCalories = dto.targetCalories;
                healthInfo.recommendIntake = dto.recommendIntake;
                healthInfo = removeUndefined(healthInfo);
                return healthInfo;
    }
}