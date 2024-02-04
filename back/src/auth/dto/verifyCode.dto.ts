import { ApiProperty } from "@nestjs/swagger";
import { IsEmail, IsNotEmpty, IsString, Length} from "class-validator";

export class VerifyCodeDto {

    @ApiProperty({example:'your email'})
    @IsNotEmpty()
    @IsEmail({},{ message: "이메일 형식이 올바르지 않습니다." })
    email: string;

    @ApiProperty({example:'sent code'})
    @IsNotEmpty()
    @IsString()
    code: string;
}