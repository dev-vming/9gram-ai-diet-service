import { ApiProperty } from "@nestjs/swagger";
import { IsEmail, IsNotEmpty, IsString } from "class-validator";

export class LocalSignupDto {
    @ApiProperty({example: 'elice', description: '유저이름'})
    @IsNotEmpty()
    @IsString()
    username: string;

    @ApiProperty({example: 'elice@gmail.com', description: '이메일'})
    @IsNotEmpty()
    @IsEmail()
    email: string;
    
    @ApiProperty({example: '1234', description: '비밀번호'})
    @IsNotEmpty()
    @IsString()
    password: string;
}