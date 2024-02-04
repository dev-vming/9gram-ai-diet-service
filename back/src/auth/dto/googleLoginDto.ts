import { IsEmail, IsNotEmpty, IsString } from "class-validator";

export class GoogleLoginDto {

    @IsNotEmpty()
    @IsString()
    displayName: string;

    @IsNotEmpty()
    @IsEmail()
    email: string;

    @IsNotEmpty()
    @IsString()
    providerId: string;

}