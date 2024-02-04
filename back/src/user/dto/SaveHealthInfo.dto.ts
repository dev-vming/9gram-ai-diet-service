import { ApiProperty } from "@nestjs/swagger"
import { IsArray, IsBoolean, IsDate, IsEnum, IsNumber, IsOptional, IsString } from "class-validator"
import { Gender } from "../utils/user.enums"
import { ActivityAmount, DietGoal } from "../utils/health-info.enums"
import { HealthInfo } from "../entities/health-info.entity"

export class SaveHealthInfoDto{

    @ApiProperty( { example: 174, description:'유저의 키' } )
    @IsNumber()
    @IsOptional()
    height: number
    
    @ApiProperty( {example: 68, description:'유저의 몸무게' } )
    @IsNumber()
    @IsOptional()
    weight: number
    
    @ApiProperty({ example: "60", description: "유저의 목표 체중" })
    @IsNumber()
    @IsOptional()
    targetWeight: number
    
    @ApiProperty( { example: 1 , description: 'MUSCLE_GAIN = 1, WEIGHT_LOSS = 2,MAINTENANCE = 3,' } )
    @IsOptional()
    dietGoal: DietGoal
    
    @ApiProperty({example: 1 , description: 'LOW = 1, MEDIUM = 2, HIGH = 3'})
    @IsOptional()
    activityAmount: ActivityAmount
    
    @ApiProperty({ example: '2000', description: '유저의 하루 권장 칼로리 섭취량'  })
    @IsNumber()
    @IsOptional()
    targetCalories: number

    @ApiProperty( { example: [200,80,50,4], description:'단탄지식이섬유별 권장 섭취량 (g) ' } )
    @IsArray()
    @IsOptional()
    recommendIntake: number[]

    mapSaveDtoToEntity(dto: SaveHealthInfoDto){
        const healthInfo = new HealthInfo();
        healthInfo.weight = dto.weight;
        healthInfo.height = dto.height;
        healthInfo.targetWeight = dto.targetWeight;
        healthInfo.dietGoal = dto.dietGoal;
        healthInfo.activityAmount = dto.activityAmount;
        healthInfo.targetCalories = dto.targetCalories;
        healthInfo.recommendIntake = dto.recommendIntake;
        return healthInfo;
    }
}