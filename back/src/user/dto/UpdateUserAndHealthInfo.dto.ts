import { ApiProperty } from "@nestjs/swagger"
import { IsArray, IsBoolean, IsDate, IsEnum, IsNumber, IsOptional, IsString } from "class-validator"
import { Gender } from "../utils/user.enums"
import { ActivityAmount, DietGoal } from "../utils/health-info.enums"

export class UpdateUserAndHealthInfoDto{

    @ApiProperty( { example: 'testUser', description: '유저의 아이디' } )
    @IsString()
    @IsOptional()
    username: string

    // @ApiProperty( { example: 'password', description: '유저의 비밀번호' } )
    // @IsString()
    // @IsOptional()
    // password: string

    @ApiProperty({ example: '1999-01-01', description: '생년월일'  } )
    @IsDate()
    @IsOptional()
    birthDay: Date

    @ApiProperty( { example: 1, description: '1:MALE, 2:FEMALE, 3:OTHER ' } )
    @IsOptional()
    gender: string

    @ApiProperty( { example: '이미지url', description:'사용자 프로필 사진 url링크' } )
    @IsString()
    @IsOptional()
    profileImage: string

    @ApiProperty( { example: true, description: '유료 멤버십 여부' } )
    @IsBoolean()
    @IsOptional()
    membership: boolean

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
    goal: DietGoal
    
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

}