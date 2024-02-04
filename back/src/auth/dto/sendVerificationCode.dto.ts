import { Expose } from 'class-transformer';
import { ApiProperty } from "@nestjs/swagger";
import { IsEmail, IsNotEmpty, IsString } from "class-validator";

export class sendVerificationCodeDto {
    @ApiProperty({example:'your email'})
    @IsNotEmpty()
    @IsEmail({},{ message: "이메일 형식이 올바르지 않습니다." })
    email: string;
}