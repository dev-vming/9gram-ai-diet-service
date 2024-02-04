import { IsDate, IsString } from "class-validator";


export class CreateImageDto {
    @IsString()
    foodImageUrl: string;
}