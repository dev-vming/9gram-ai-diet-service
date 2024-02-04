import { ApiProperty } from "@nestjs/swagger";
import { IsEmail, IsNotEmpty, IsString } from "class-validator";


export class LocalLoginDto {
    @ApiProperty({example: 'elice@gmail.com', description: '이메일'})
    @IsEmail()
    @IsNotEmpty()
    email: string;

    @ApiProperty({example: '1234', description: '비밀번호'})
    @IsString()
    @IsNotEmpty()
    password: string;
}